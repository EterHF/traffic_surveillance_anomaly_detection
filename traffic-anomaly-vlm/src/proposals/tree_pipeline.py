from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from src.evidence.builder import EvidenceBuilder
from src.schemas import EvidencePack, SpanProposal, TrackObject
from src.vlm.parser import (
    parse_span_score_output,
    parse_span_score_output_for_stage1,
    parse_span_score_output_for_stage2,
)
from src.vlm.prompts import build_span_score_prompt, build_stage1_prompt, build_stage2_prompt

if TYPE_CHECKING:
    from src.vlm.model_loader import LocalVLM


def refine_spans_with_vlm(
    *,
    vlm: LocalVLM,
    frame_spans: list[tuple[int, int]],
    images: list[str],
    frame_ids: list[int],
    all_tracks: list[TrackObject],
    eventness_scores: list[float] | np.ndarray | None = None,
    method: Literal["montage", "frames", "enhanced"] = "montage",
    output_assets_dir: str = "./assets/evidence",
    output_dir: str | None = None,
    prompt_method: str = "single_stage",
    min_confidence: float = 0.35,
    positive_threshold: float = 0.5,
    merge_gap: int = 0,
) -> list[SpanProposal]:
    """Score frame-id spans with VLM and return merged positive spans."""

    scores = _prepare_scores(eventness_scores, len(frame_ids))
    evidence_builder = EvidenceBuilder(output_assets_dir=output_assets_dir)
    normalized_spans = _normalize_frame_spans(frame_spans)
    proposals = [
        _proposal_from_frame_span(
            frame_span=frame_span,
            span_id=f"span.{idx}",
            frame_ids=frame_ids,
            scores=scores,
            left_gap=_left_gap(normalized_spans, idx - 1),
            right_gap=_right_gap(normalized_spans, idx - 1),
        )
        for idx, frame_span in enumerate(normalized_spans, start=1)
    ]

    for proposal in proposals:
        evidence = evidence_builder.build(
            proposal=proposal,
            images=images,
            frame_ids=frame_ids,
            all_tracks=all_tracks,
            method=method,
            output_dir=output_dir,
        )
        _score_proposal_with_vlm(
            proposal,
            evidence,
            vlm=vlm,
            prompt_method=prompt_method,
            min_confidence=min_confidence,
        )
        proposal.fused_score = _decision_score(proposal)
        proposal.is_positive = proposal.fused_score >= float(positive_threshold)

    return merge_positive_spans(proposals, gap=merge_gap)


def merge_positive_spans(spans: list[SpanProposal], gap: int) -> list[SpanProposal]:
    """Merge positive spans and expand boundaries across small neighboring gaps."""

    ordered = sorted(spans, key=lambda item: (int(item.start_frame), int(item.end_frame)))
    merged: list[SpanProposal] = []
    group: list[SpanProposal] = []
    gap = max(0, int(gap))

    for span in ordered:
        if not span.is_positive:
            if group:
                merged.append(_merge_group(group, len(merged) + 1))
                group = []
            continue

        if group and gap > 0 and int(span.start_frame) - int(group[-1].end_frame) > gap:
            merged.append(_merge_group(group, len(merged) + 1))
            group = []
        group.append(span)

    if group:
        merged.append(_merge_group(group, len(merged) + 1))

    return [_expand_boundaries(span, max_gap=gap) for span in merged]


def _prepare_scores(
    eventness_scores: list[float] | np.ndarray | None,
    expected_len: int,
) -> np.ndarray:
    if eventness_scores is None:
        return np.zeros((expected_len,), dtype=np.float32)
    scores = np.asarray(eventness_scores, dtype=np.float32)
    if scores.size != expected_len:
        raise ValueError("frame_ids and eventness_scores must have the same length")
    return scores


def _normalize_frame_spans(frame_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    for start_frame, end_frame in frame_spans:
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        normalized.append((start_frame, end_frame))
    return sorted(normalized, key=lambda item: (item[0], item[1]))


def _left_gap(spans: list[tuple[int, int]], index: int) -> int:
    if index <= 0:
        return 0
    return max(0, int(spans[index][0]) - int(spans[index - 1][1]))


def _right_gap(spans: list[tuple[int, int]], index: int) -> int:
    if index + 1 >= len(spans):
        return 0
    return max(0, int(spans[index + 1][0]) - int(spans[index][1]))


def _expand_boundaries(span: SpanProposal, max_gap: int) -> SpanProposal:
    left_expand = _gap_extension(span.left_gap, max_gap)
    right_expand = _gap_extension(span.right_gap, max_gap)
    span.start_frame = max(0, int(span.start_frame) - left_expand)
    span.end_frame = int(span.end_frame) + right_expand
    return span


def _gap_extension(gap: int, max_gap: int) -> int:
    gap = max(0, int(gap))
    max_gap = max(0, int(max_gap))
    if gap == 0 or max_gap == 0 or gap > max_gap:
        return 0
    return gap // 2


def _proposal_from_frame_span(
    *,
    frame_span: tuple[int, int],
    span_id: str,
    frame_ids: list[int],
    scores: np.ndarray,
    left_gap: int,
    right_gap: int,
) -> SpanProposal:
    start_frame, end_frame = int(frame_span[0]), int(frame_span[1])
    if end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame

    frame_to_idx = {int(frame_id): idx for idx, frame_id in enumerate(frame_ids)}
    if start_frame not in frame_to_idx or end_frame not in frame_to_idx:
        raise ValueError(f"frame span {frame_span} is not aligned to sampled frame_ids")

    start_idx = int(frame_to_idx[start_frame])
    end_idx = int(frame_to_idx[end_frame])
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
        start_frame, end_frame = end_frame, start_frame

    local = scores[start_idx : end_idx + 1]
    rel_peak = int(np.argmax(local)) if local.size else 0
    peak_idx = int(start_idx + rel_peak)
    eventness_peak = float(local[rel_peak]) if local.size else 0.0
    eventness_mean = float(np.mean(local)) if local.size else 0.0
    return SpanProposal(
        span_id=str(span_id),
        start_idx=start_idx,
        end_idx=end_idx,
        peak_idx=peak_idx,
        start_frame=start_frame,
        end_frame=end_frame,
        peak_frame=int(frame_ids[peak_idx]),
        eventness_peak=eventness_peak,
        eventness_mean=eventness_mean,
        left_gap=int(left_gap),
        right_gap=int(right_gap),
        prior_score=0.7 * eventness_peak + 0.3 * eventness_mean,
    )


def _score_proposal_with_vlm(
    proposal: SpanProposal,
    evidence: EvidencePack,
    *,
    vlm: LocalVLM,
    prompt_method: str,
    min_confidence: float,
) -> None:
    from src.vlm.infer import run_inference

    prior = _clip01(proposal.prior_score)
    if prompt_method == "two_stage":
        stage1_prompt = build_stage1_prompt(
            evidence.summary,
            evidence_method=str(evidence.summary.get("evidence_method", "montage")),
        )
        stage1_raw = run_inference(vlm, stage1_prompt, evidence.keyframe_paths)
        stage1_output = parse_span_score_output_for_stage1(stage1_raw)
        stage2_prompt = build_stage2_prompt(
            stage1_output,
            summary=evidence.summary,
            evidence_method=str(evidence.summary.get("evidence_method", "montage")),
        )
        stage2_raw = run_inference(vlm, stage2_prompt, evidence.keyframe_paths)
        parsed = parse_span_score_output_for_stage2(stage2_raw)
    else:
        prompt = build_span_score_prompt(
            evidence.summary,
            evidence_method=str(evidence.summary.get("evidence_method", "montage")),
        )
        raw = run_inference(vlm, prompt, evidence.keyframe_paths)
        parsed = parse_span_score_output(raw)

    confidence = _clip01(float(parsed.get("confidence", 0.0)))
    proposal.vlm_confidence = confidence
    if confidence < float(min_confidence):
        proposal.vlm_score = prior
        return

    proposal.vlm_score = _clip01(float(parsed.get("anomaly_score", prior)))


def _decision_score(proposal: SpanProposal) -> float:
    prior = _clip01(proposal.prior_score)
    vlm_score = _clip01(proposal.vlm_score if proposal.vlm_score > 0.0 else prior)
    confidence = _clip01(proposal.vlm_confidence)
    return _clip01((1.0 - confidence) * prior + confidence * vlm_score)


def _merge_group(group: list[SpanProposal], group_index: int) -> SpanProposal:
    first = group[0]
    last = group[-1]
    best = max(
        group,
        key=lambda item: (
            float(item.fused_score),
            float(item.vlm_score),
            float(item.prior_score),
            float(item.eventness_peak),
        ),
    )
    return SpanProposal(
        span_id=f"merged.{group_index}",
        start_idx=int(first.start_idx),
        end_idx=int(last.end_idx),
        peak_idx=int(best.peak_idx),
        start_frame=int(first.start_frame),
        end_frame=int(last.end_frame),
        peak_frame=int(best.peak_frame),
        eventness_peak=max(float(item.eventness_peak) for item in group),
        eventness_mean=float(np.mean([float(item.eventness_mean) for item in group])),
        left_gap=int(first.left_gap),
        right_gap=int(last.right_gap),
        prior_score=max(float(item.prior_score) for item in group),
        vlm_score=max(float(item.vlm_score) for item in group),
        vlm_confidence=max(float(item.vlm_confidence) for item in group),
        fused_score=max(float(item.fused_score) for item in group),
        is_positive=True,
    )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
